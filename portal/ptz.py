"""PTZ control for Hikvision cameras + a camera-namespaced local preset store.

The camera(s) expose Hikvision ISAPI over HTTP(S) + HTTP Digest. This module is
used inside the portal container (async, httpx) and is also runnable as a CLI
for local verification:

    python -m portal.ptz --camera cam08 health
    python -m portal.ptz --camera cam08 pan --speed 20 --ms 500
    python -m portal.ptz tilt --speed -20 --ms 500
    python -m portal.ptz stop
    python -m portal.ptz presets list
    python -m portal.ptz presets add home --description "Home view"
    python -m portal.ptz presets goto home
    python -m portal.ptz presets resave home
    python -m portal.ptz presets rename home gate
    python -m portal.ptz presets describe gate --description "Gate view"
    python -m portal.ptz presets delete gate

Preset model (HYBRID): the camera can move and store preset positions, but this
model has NO absolute-position feedback (`/status` always reports zeros and
`/absoluteEx` is unsupported). The physical position therefore lives in a camera
preset slot; `PTZ_PRESETS_FILE` (default `/media/portal/ptz_presets.json`) is the
source of truth for the preset NAME + description + camera slot id, namespaced
per camera.

Portal movement uses ONVIF ContinuousMove with a camera-enforced Timeout
(portal/onvif.py); the ISAPI /momentary path remains only as an explicit
PTZ_MOVE_BACKEND=isapi fallback.

Credentials come from the environment, never this file.
ISAPI presets (per-camera first):
    PTZ_USER_<CAM> / PTZ_PASS_<CAM>
    PTZ_USER / PTZ_PASS
    else FRIGATE_HIK_RTSP_USER / FRIGATE_HIK_RTSP_PASS
    else FRIGATE_RTSP_USER / FRIGATE_RTSP_PASS
ONVIF movement (per-camera first):
    PTZ_ONVIF_USER_<CAM> / PTZ_ONVIF_PASS_<CAM>
    PTZ_ONVIF_USER / PTZ_ONVIF_PASS
    else ONVIF_USER / ONVIF_PASS
Camera target/tuning comes from portal.conf (PTZ_* keys, see config/portal.conf).
PTZ_CAMERA is comma-separated; the portal resolves each host from Frigate
`/api/config` go2rtc.streams.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import copy
import fcntl
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape

import httpx

from portal import config as pconf
from portal import frigate
from portal import onvif as onvif_api

NS = "http://www.hikvision.com/ver20/XMLSchema"
DEFAULT_CONF = os.environ.get("PORTAL_CONF", "/config/portal.conf")
DEFAULT_PRESETS_FILE = "/media/portal/ptz_presets.json"


class PTZError(RuntimeError):
    """Base error for PTZ/camera/preset failures."""


class PTZAuthError(PTZError):
    """The camera rejected the configured credentials."""


class PTZHTTPError(PTZError):
    """The camera returned an unexpected HTTP status."""

    def __init__(self, status_code: int, detail: str = ""):
        self.status_code = status_code
        super().__init__(f"camera HTTP {status_code}" + (f": {detail}" if detail else ""))


class PresetStoreError(PTZError):
    """The local preset file is missing, corrupt, or cannot be written."""


class PresetNotFound(PTZError):
    """No preset with that name exists in the local store."""


class PresetExists(PTZError):
    """A preset with that name already exists in the local store."""


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child_text(el: ET.Element, name: str) -> str:
    for child in el:
        if _local(child.tag) == name:
            return (child.text or "").strip()
    return ""


def _child_int(el: ET.Element, name: str) -> int | None:
    value = _child_text(el, name)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _status_detail(text: str) -> str:
    """Extract a readable Hikvision ResponseStatus detail, if present."""
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return text.strip()[:300]
    bits = []
    for name in ("statusString", "subStatusCode", "description", "errorMsg"):
        for el in root.iter():
            if _local(el.tag) == name and (el.text or "").strip():
                bits.append((el.text or "").strip())
                break
    return " / ".join(dict.fromkeys(bits))[:400] or text.strip()[:300]


def _status_ok(text: str) -> bool:
    """True for Hikvision ResponseStatus bodies with statusCode 1/OK."""
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return True  # some endpoints return an empty body on success
    code = None
    for el in root.iter():
        if _local(el.tag) == "statusCode":
            code = (el.text or "").strip()
            break
    return code in (None, "1")


class PTZClient:
    """Minimal async Hikvision ISAPI PTZ client."""

    def __init__(
        self,
        host: str,
        user: str,
        password: str,
        *,
        port: int = 80,
        scheme: str = "http",
        channel: int = 1,
        verify_tls: bool = False,
        timeout: float = 8.0,
    ):
        if not host:
            raise PTZError("PTZ_HOST is empty")
        if not user or not password:
            raise PTZError(
                "camera credentials are missing (set PTZ_USER/PTZ_PASS or "
                "FRIGATE_HIK_RTSP_USER/FRIGATE_HIK_RTSP_PASS)"
            )
        self.host = host
        self.port = int(port)
        self.scheme = scheme
        self.channel = int(channel)
        self.timeout = float(timeout)
        self.base = self._make_base()
        self._client = httpx.AsyncClient(
            auth=httpx.DigestAuth(user, password),
            verify=verify_tls,
            timeout=timeout,
            follow_redirects=False,
        )

    def _make_base(self) -> str:
        host = self.host
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        default_port = 443 if self.scheme == "https" else 80
        suffix = "" if self.port == default_port else f":{self.port}"
        return f"{self.scheme}://{host}{suffix}"

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "PTZClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    def _url(self, path: str) -> str:
        return self.base + path

    def _ptz_path(self, suffix: str) -> str:
        return f"/ISAPI/PTZCtrl/channels/{self.channel}{suffix}"

    async def _request(
        self,
        method: str,
        path: str,
        *,
        content: bytes | None = None,
        expect: tuple[int, ...] = (200,),
    ) -> httpx.Response:
        headers = {}
        if content is not None:
            headers["Content-Type"] = "application/xml; charset=utf-8"
        try:
            resp = await self._client.request(
                method, self._url(path), content=content, headers=headers
            )
        except httpx.HTTPError as exc:
            raise PTZError(f"camera unreachable: {exc}") from exc
        if resp.status_code == 401:
            raise PTZAuthError("camera rejected the configured credentials (401)")
        if resp.status_code not in expect:
            raise PTZHTTPError(resp.status_code, _status_detail(resp.text))
        return resp

    async def device_info(self) -> dict:
        resp = await self._request("GET", "/ISAPI/System/deviceInfo")
        root = ET.fromstring(resp.text)
        return {
            key: _child_text(root, key)
            for key in ("deviceName", "model", "serialNumber",
                        "firmwareVersion", "firmwareReleasedDate")
        }

    async def capabilities(self) -> dict:
        resp = await self._request("GET", self._ptz_path("/capabilities"))
        root = ET.fromstring(resp.text)
        max_presets = _child_int(root, "maxPresetNum")
        return {
            "max_presets": max_presets if max_presets is not None else 300,
            "raw": resp.text,
        }

    async def max_presets(self) -> int:
        return (await self.capabilities())["max_presets"]

    async def camera_presets(self) -> dict[int, dict]:
        """Return {preset_id: {enabled, name}} for presets configured on the camera."""
        resp = await self._request("GET", self._ptz_path("/presets"))
        root = ET.fromstring(resp.text)
        presets: dict[int, dict] = {}
        for el in root.iter():
            if _local(el.tag) != "PTZPreset":
                continue
            pid = _child_int(el, "id")
            if pid is None:
                continue
            presets[pid] = {
                "enabled": _child_text(el, "enabled").lower() == "true",
                "name": _child_text(el, "presetName"),
            }
        return presets

    @staticmethod
    def _continuous_xml(pan: int, tilt: int, zoom: int) -> bytes:
        return (
            '<?xml version="1.0" encoding="UTF-8"?>'
            f'<PTZData version="2.0" xmlns="{NS}">'
            f"<pan>{int(pan)}</pan><tilt>{int(tilt)}</tilt>"
            f"<zoom>{int(zoom)}</zoom></PTZData>"
        ).encode("utf-8")

    @staticmethod
    def _momentary_xml(pan: int, tilt: int, zoom: int, duration_ms: int) -> bytes:
        return (
            '<?xml version="1.0" encoding="UTF-8"?>'
            f'<PTZData version="2.0" xmlns="{NS}">'
            f"<pan>{int(pan)}</pan><tilt>{int(tilt)}</tilt>"
            f"<zoom>{int(zoom)}</zoom>"
            f"<Momentary><duration>{int(duration_ms)}</duration></Momentary>"
            "</PTZData>"
        ).encode("utf-8")

    @staticmethod
    def _check_speed(pan: int, tilt: int, zoom: int) -> None:
        for label, value in (("pan", pan), ("tilt", tilt), ("zoom", zoom)):
            if not -100 <= int(value) <= 100:
                raise PTZError(f"{label} must be between -100 and 100")

    async def continuous(self, pan: int, tilt: int = 0, zoom: int = 0) -> None:
        self._check_speed(pan, tilt, zoom)
        resp = await self._request(
            "PUT", self._ptz_path("/continuous"),
            content=self._continuous_xml(pan, tilt, zoom),
        )
        if not _status_ok(resp.text):
            raise PTZHTTPError(resp.status_code, _status_detail(resp.text))

    async def momentary(
        self, pan: int, tilt: int = 0, zoom: int = 0, duration_ms: int = 400
    ) -> None:
        """Move for a bounded duration; the camera is expected to stop itself."""
        self._check_speed(pan, tilt, zoom)
        if not 50 <= int(duration_ms) <= 10_000:
            raise PTZError("momentary duration must be between 50 and 10000 ms")
        resp = await self._request(
            "PUT", self._ptz_path("/momentary"),
            content=self._momentary_xml(pan, tilt, zoom, duration_ms),
        )
        if not _status_ok(resp.text):
            raise PTZHTTPError(resp.status_code, _status_detail(resp.text))

    async def stop(self) -> None:
        await self.continuous(0, 0, 0)

    async def move(self, pan: int, tilt: int = 0, duration_ms: int = 500) -> None:
        """Move continuously for a bounded time, always stopping afterwards."""
        if not 50 <= int(duration_ms) <= 10_000:
            raise PTZError("duration must be between 50 and 10000 ms")
        await self.continuous(pan, tilt, 0)
        try:
            await asyncio.sleep(int(duration_ms) / 1000.0)
        finally:
            with contextlib.suppress(Exception):
                await self.stop()

    @staticmethod
    def _preset_xml(preset_id: int, name: str) -> bytes:
        safe_name = escape(name[:64])
        return (
            '<?xml version="1.0" encoding="UTF-8"?>'
            f'<PTZPreset version="2.0" xmlns="{NS}">'
            f"<enabled>true</enabled><id>{int(preset_id)}</id>"
            f"<presetName>{safe_name}</presetName></PTZPreset>"
        ).encode("utf-8")

    async def save_preset(self, preset_id: int, name: str) -> None:
        """Store the CURRENT camera view into a camera preset slot."""
        resp = await self._request(
            "PUT", self._ptz_path(f"/presets/{int(preset_id)}"),
            content=self._preset_xml(preset_id, name),
        )
        if not _status_ok(resp.text):
            raise PTZHTTPError(resp.status_code, _status_detail(resp.text))

    async def delete_preset(self, preset_id: int) -> bool:
        """Delete a camera preset slot. Returns False if it did not exist."""
        resp = await self._request(
            "DELETE", self._ptz_path(f"/presets/{int(preset_id)}"),
            expect=(200, 404),
        )
        if resp.status_code == 404:
            return False
        if not _status_ok(resp.text):
            raise PTZHTTPError(resp.status_code, _status_detail(resp.text))
        return True

    async def goto_preset(self, preset_id: int) -> None:
        resp = await self._request(
            "PUT", self._ptz_path(f"/presets/{int(preset_id)}/goto"),
            content=b"",
        )
        if not _status_ok(resp.text):
            raise PTZHTTPError(resp.status_code, _status_detail(resp.text))

    async def health(self) -> dict:
        info = await self.device_info()
        caps = await self.capabilities()
        try:
            presets = await self.camera_presets()
            camera_preset_ids = sorted(presets)
        except PTZError:
            camera_preset_ids = None
        return {
            "host": self.host,
            "channel": self.channel,
            "device": info,
            "max_presets": caps["max_presets"],
            "camera_preset_ids": camera_preset_ids,
        }


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _validate_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        raise PTZError("preset name is empty")
    if len(name) > 64:
        raise PTZError("preset name must be at most 64 characters")
    if any(ord(ch) < 32 for ch in name):
        raise PTZError("preset name contains control characters")
    return name


class PresetStore:
    """Atomic JSON store for external preset metadata, namespaced by camera.

    Layout (v2):
        {
          "version": 2,
          "cameras": {
            "cam08": {
              "presets": {"<name>": {"camera_preset_id": 20, ...}}
            }
          }
        }

    A legacy v1 flat file ({"version":1,"presets":{...}}) is migrated in memory
    to the `legacy_camera` bucket on first read.
    """

    def __init__(
        self, path: str = DEFAULT_PRESETS_FILE, legacy_camera: str = "cam08"
    ):
        self.path = Path(path)
        self.legacy_camera = legacy_camera

    @contextlib.contextmanager
    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = Path(str(self.path) + ".lock")
        with open(lock_path, "a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _validate_presets(path: Path, camera: str, presets: dict) -> None:
        if not isinstance(presets, dict):
            raise PresetStoreError(f"{path}: camera {camera!r} presets must be an object")
        for name, entry in presets.items():
            if not isinstance(entry, dict) or not isinstance(
                entry.get("camera_preset_id"), int
            ):
                raise PresetStoreError(
                    f"{path}: preset {camera}/{name!r} has no integer camera_preset_id"
                )

    def _read_unlocked(self) -> dict:
        if not self.path.exists():
            return {"version": 2, "cameras": {}}
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError) as exc:
            raise PresetStoreError(f"cannot read {self.path}: {exc}") from exc
        if not isinstance(data, dict):
            raise PresetStoreError(f"{self.path}: root must be a JSON object")

        if "cameras" in data:
            cameras = data.get("cameras")
            if not isinstance(cameras, dict):
                raise PresetStoreError(f"{self.path}: 'cameras' must be an object")
            for camera, bucket in cameras.items():
                if not isinstance(bucket, dict):
                    raise PresetStoreError(
                        f"{self.path}: camera {camera!r} must be an object"
                    )
                self._validate_presets(self.path, camera, bucket.get("presets", {}))
            return {"version": 2, "cameras": cameras}

        # Legacy v1 flat layout -> v2 in memory.
        presets = data.get("presets")
        if not isinstance(presets, dict):
            raise PresetStoreError(f"{self.path}: 'presets' must be an object")
        self._validate_presets(self.path, self.legacy_camera, presets)
        return {
            "version": 2,
            "cameras": {self.legacy_camera: {"presets": presets}},
        }

    def _write_unlocked(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)
        fd, tmp = tempfile.mkstemp(prefix=self.path.name + ".", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(tmp, 0o644)
            os.replace(tmp, self.path)
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    @staticmethod
    def _bucket(data: dict, camera: str) -> dict:
        cameras = data.setdefault("cameras", {})
        bucket = cameras.setdefault(camera, {})
        bucket.setdefault("presets", {})
        return bucket["presets"]

    @contextlib.contextmanager
    def transaction(self, camera: str):
        with self._locked():
            data = self._read_unlocked()
            presets = self._bucket(data, camera)
            yield presets
            self._write_unlocked(data)

    def read(self, camera: str) -> dict:
        with self._locked():
            data = self._read_unlocked()
            return copy.deepcopy(self._bucket(data, camera))

    def read_all(self) -> dict:
        with self._locked():
            return copy.deepcopy(self._read_unlocked())


def _alloc_preset_id(
    used: set[int], camera_presets: dict[int, dict], max_presets: int
) -> int:
    occupied = set(used)
    occupied.update(pid for pid, info in camera_presets.items() if info.get("enabled"))
    for pid in range(1, max_presets + 1):
        if pid not in occupied:
            return pid
    raise PTZError(f"no free camera preset slot (max {max_presets})")


class PTZManager:
    """High-level PTZ + external-preset operations for ONE camera."""

    def __init__(self, client: PTZClient, store: PresetStore, camera: str = "cam08"):
        self.client = client
        self.store = store
        self.camera = camera

    async def list_presets(self, include_camera: bool = True) -> list[dict]:
        data = self.store.read(self.camera)
        camera_presets = None
        if include_camera:
            try:
                camera_presets = await self.client.camera_presets()
            except PTZError:
                camera_presets = None
        items = []
        for name in sorted(data):
            entry = data[name]
            pid = entry["camera_preset_id"]
            items.append({
                "camera": self.camera,
                "name": name,
                "camera_preset_id": pid,
                "description": entry.get("description", ""),
                "created_at": entry.get("created_at", ""),
                "updated_at": entry.get("updated_at", ""),
                "camera_present": (
                    pid in camera_presets if camera_presets is not None else None
                ),
            })
        return items

    async def add(
        self, name: str, description: str = "", camera_preset_id: int | None = None
    ) -> dict:
        name = _validate_name(name)
        saved_id = None
        try:
            with self.store.transaction(self.camera) as presets:
                if name in presets:
                    raise PresetExists(f"preset {name!r} already exists")
                camera = await self.client.camera_presets()
                max_presets = await self.client.max_presets()

                if camera_preset_id is None:
                    pid = _alloc_preset_id(
                        {e["camera_preset_id"] for e in presets.values()},
                        camera,
                        max_presets,
                    )
                else:
                    pid = int(camera_preset_id)
                    if not 1 <= pid <= max_presets:
                        raise PTZError(
                            f"camera preset id must be 1..{max_presets}"
                        )
                    if pid in {e["camera_preset_id"] for e in presets.values()}:
                        raise PresetExists(
                            f"camera preset {pid} is already managed by a local preset"
                        )
                    if camera.get(pid, {}).get("enabled"):
                        raise PresetExists(
                            f"camera preset {pid} is already configured on the camera"
                        )

                await self.client.save_preset(pid, name)
                saved_id = pid
                now = _now()
                presets[name] = {
                    "camera_preset_id": pid,
                    "description": description or "",
                    "created_at": now,
                    "updated_at": now,
                }
                return {
                    "camera": self.camera,
                    "name": name,
                    "camera_preset_id": pid,
                    "description": description or "",
                    "created_at": now,
                    "updated_at": now,
                }
        except Exception:
            if saved_id is not None:
                with contextlib.suppress(Exception):
                    await self.client.delete_preset(saved_id)
            raise

    async def goto(self, name: str) -> dict:
        name = _validate_name(name)
        data = self.store.read(self.camera)
        entry = data.get(name)
        if entry is None:
            raise PresetNotFound(f"preset {name!r} does not exist")
        pid = entry["camera_preset_id"]
        await self.client.goto_preset(pid)
        return {"camera": self.camera, "name": name, "camera_preset_id": pid}

    async def resave(self, name: str) -> dict:
        """Overwrite a camera preset slot with the camera's CURRENT view."""
        name = _validate_name(name)
        with self.store.transaction(self.camera) as presets:
            entry = presets.get(name)
            if entry is None:
                raise PresetNotFound(f"preset {name!r} does not exist")
            pid = entry["camera_preset_id"]
            await self.client.save_preset(pid, name)
            entry["updated_at"] = _now()
            return {"camera": self.camera, "name": name,
                    "camera_preset_id": pid, "updated_at": entry["updated_at"]}

    async def rename(self, old: str, new: str) -> dict:
        old = _validate_name(old)
        new = _validate_name(new)
        with self.store.transaction(self.camera) as presets:
            entry = presets.get(old)
            if entry is None:
                raise PresetNotFound(f"preset {old!r} does not exist")
            if new in presets:
                raise PresetExists(f"preset {new!r} already exists")
            presets[new] = entry
            del presets[old]
            entry["updated_at"] = _now()
            return {"camera": self.camera, "name": new,
                    "camera_preset_id": entry["camera_preset_id"],
                    "updated_at": entry["updated_at"]}

    async def describe(self, name: str, description: str) -> dict:
        name = _validate_name(name)
        with self.store.transaction(self.camera) as presets:
            entry = presets.get(name)
            if entry is None:
                raise PresetNotFound(f"preset {name!r} does not exist")
            entry["description"] = description or ""
            entry["updated_at"] = _now()
            return {"camera": self.camera, "name": name,
                    "description": entry["description"],
                    "updated_at": entry["updated_at"]}

    async def delete(self, name: str) -> dict:
        name = _validate_name(name)
        with self.store.transaction(self.camera) as presets:
            entry = presets.get(name)
            if entry is None:
                raise PresetNotFound(f"preset {name!r} does not exist")
            pid = entry["camera_preset_id"]
            deleted_on_camera = await self.client.delete_preset(pid)
            del presets[name]
            return {"camera": self.camera, "name": name,
                    "camera_preset_id": pid,
                    "camera_deleted": deleted_on_camera}


def _camera_key(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").upper()


def _env_first(*names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return ""


def _credentials_for(camera: str) -> tuple[str, str]:
    key = _camera_key(camera)
    user = _env_first(
        f"PTZ_USER_{key}", "PTZ_USER", "FRIGATE_HIK_RTSP_USER",
        "FRIGATE_RTSP_USER",
    )
    password = _env_first(
        f"PTZ_PASS_{key}", "PTZ_PASS", "FRIGATE_HIK_RTSP_PASS",
        "FRIGATE_RTSP_PASS",
    )
    return user, password


def _onvif_credentials_for(camera: str) -> tuple[str, str]:
    key = _camera_key(camera)
    user = _env_first(
        f"PTZ_ONVIF_USER_{key}", "PTZ_ONVIF_USER", "ONVIF_USER",
    )
    password = _env_first(
        f"PTZ_ONVIF_PASS_{key}", "PTZ_ONVIF_PASS", "ONVIF_PASS",
    )
    return user, password


def _host_from_rtsp(url: str | None) -> str | None:
    if not url or "://" not in url or url.startswith("ffmpeg:"):
        return None
    try:
        host = urlsplit(url).hostname
    except ValueError:
        return None
    if not host or host in ("127.0.0.1", "localhost"):
        return None
    return host


def _camera_names(cfg: dict) -> list[str]:
    raw = pconf.get(cfg, "PTZ_CAMERA", "cam08")
    names = []
    for item in str(raw).split(","):
        name = item.strip()
        if name and name not in names:
            names.append(name)
    return names


class PTZService:
    """One camera's ISAPI preset client + ONVIF movement client."""

    def __init__(
        self,
        client: PTZClient,
        manager: PTZManager,
        *,
        onvif_client: onvif_api.ONVIFClient | None = None,
        move_backend: str = "onvif",
        hold_max_s: float = 2.0,
        momentary_ms: int = 400,
        onvif_timeout_s: int = 1,
        onvif_gap_s: float = 0.0,
    ):
        self.client = client
        self.manager = manager
        self.camera = manager.camera
        self.onvif = onvif_client
        self.backend = move_backend if move_backend in (
            "onvif", "isapi", "momentary") else "onvif"
        self.hold_max_s = max(0.5, float(hold_max_s))
        self.momentary_ms = int(momentary_ms)
        self.onvif_timeout_s = max(1, int(round(float(onvif_timeout_s))))
        self.onvif_gap_s = max(0.0, float(onvif_gap_s))
        self._lock = asyncio.Lock()
        self._stop_task: asyncio.Task | None = None
        self._move_seq = 0
        self._last_onvif_move_at = 0.0

    @property
    def move_backend(self) -> str:
        return self.backend

    @property
    def move_interval_ms(self) -> int:
        if self.backend == "onvif":
            return max(200, int(
                (self.onvif_timeout_s + self.onvif_gap_s) * 1000))
        return 220

    def _cancel_stop_task(self) -> None:
        if self._stop_task is not None:
            self._stop_task.cancel()
            self._stop_task = None

    def _schedule_stop(self) -> None:
        self._cancel_stop_task()
        self._move_seq += 1
        self._stop_task = asyncio.create_task(self._auto_stop(self._move_seq))

    async def _auto_stop(self, seq: int) -> None:
        try:
            await asyncio.sleep(self.hold_max_s)
        except asyncio.CancelledError:
            return
        if seq != self._move_seq:
            return
        with contextlib.suppress(Exception):
            await self.client.stop()

    async def move(self, pan: int, tilt: int = 0, zoom: int = 0) -> None:
        if self.backend == "onvif":
            if self.onvif is None:
                raise PTZError(
                    "ONVIF credentials are not configured "
                    "(set PTZ_ONVIF_USER/PTZ_ONVIF_PASS)")
            async with self._lock:
                now = time.monotonic()
                remaining = (
                    self.onvif_timeout_s
                    - (now - self._last_onvif_move_at)
                )
                if self._last_onvif_move_at and remaining > 0.05:
                    raise PTZError(
                        "previous ONVIF pulse is still active; "
                        f"wait {remaining:.2f}s")
                await self.onvif.continuous(
                    pan, tilt, zoom, timeout_s=self.onvif_timeout_s)
                self._last_onvif_move_at = time.monotonic()
            return
        async with self._lock:
            await self.client.momentary(
                pan, tilt, zoom, duration_ms=self.momentary_ms
            )
            self._schedule_stop()

    async def stop(self) -> None:
        async with self._lock:
            self._move_seq += 1
            self._last_onvif_move_at = 0.0
            self._cancel_stop_task()
            if self.backend == "onvif" and self.onvif is not None:
                try:
                    await self.onvif.stop()
                    return
                except Exception:
                    # Fall back to ISAPI stop; if that also fails, its error
                    # propagates.
                    await self.client.stop()
                    return
            await self.client.stop()

    async def list_presets(self, include_camera: bool = True) -> list[dict]:
        return await self.manager.list_presets(include_camera=include_camera)

    async def add_preset(
        self, name: str, description: str = "", camera_preset_id: int | None = None
    ) -> dict:
        return await self.manager.add(name, description, camera_preset_id)

    async def goto_preset(self, name: str) -> dict:
        return await self.manager.goto(name)

    async def resave_preset(self, name: str) -> dict:
        return await self.manager.resave(name)

    async def delete_preset(self, name: str) -> dict:
        return await self.manager.delete(name)

    async def aclose(self) -> None:
        self._move_seq += 1
        self._cancel_stop_task()
        if self.backend == "onvif" and self.onvif is not None:
            with contextlib.suppress(Exception):
                await self.onvif.stop()
        with contextlib.suppress(Exception):
            await self.client.stop()
        if self.onvif is not None:
            await self.onvif.aclose()
        await self.client.aclose()


class PTZRegistry:
    """Resolve configured camera names to hosts and hold one PTZService each.

    Host resolution:
      * a single PTZ_CAMERA + non-empty PTZ_HOST -> use PTZ_HOST;
      * otherwise read Frigate's /api/config go2rtc.streams.<camera> and take
        the RTSP hostname (ISAPI is then reached on PTZ_PORT, default 80).
    """

    def __init__(
        self,
        cfg: dict,
        *,
        http_client: httpx.AsyncClient | None = None,
        frigate_base: str = "http://frigate:5000",
    ):
        self.cfg = cfg
        self._http_client = http_client
        self._frigate_base = frigate_base
        self.cameras = _camera_names(cfg)
        self.store = PresetStore(
            pconf.get(cfg, "PTZ_PRESETS_FILE", DEFAULT_PRESETS_FILE),
            legacy_camera=self.cameras[0] if self.cameras else "cam08",
        )
        self._services: dict[str, PTZService] = {}
        self._resolved = False
        self._lock = asyncio.Lock()

    async def _resolve_hosts(self) -> dict[str, str]:
        override = pconf.get(self.cfg, "PTZ_HOST", "").strip()
        if len(self.cameras) == 1 and override:
            return {self.cameras[0]: override}
        if not self.cameras:
            return {}
        if self._http_client is None:
            raise PTZError("cannot resolve PTZ hosts without a Frigate HTTP client")
        config = await frigate.fetch_json(
            self._http_client, self._frigate_base, "/api/config"
        )
        streams = (config.get("go2rtc") or {}).get("streams") or {}
        hosts = {}
        for name in self.cameras:
            if not frigate.valid_camera_name(name):
                print(f"WARNING: PTZ camera name {name!r} is invalid", flush=True)
                continue
            host = _host_from_rtsp(streams.get(name))
            if host:
                hosts[name] = host
            else:
                print(
                    f"WARNING: PTZ camera {name!r} has no usable go2rtc stream host",
                    flush=True,
                )
        return hosts

    def _build_service(self, camera: str, host: str) -> PTZService:
        user, password = _credentials_for(camera)
        key = _camera_key(camera)
        scheme = pconf.get(self.cfg, "PTZ_SCHEME", "http").strip().lower()
        if scheme not in ("http", "https"):
            scheme = "http"
        port = pconf.geti(
            self.cfg, f"PTZ_PORT_{key}", pconf.geti(self.cfg, "PTZ_PORT", 80)
        )
        verify_tls = pconf.getb(self.cfg, "PTZ_VERIFY_TLS", False)
        client = PTZClient(
            host,
            user,
            password,
            port=port,
            scheme=scheme,
            channel=pconf.geti(
                self.cfg, f"PTZ_CHANNEL_{key}",
                pconf.geti(self.cfg, "PTZ_CHANNEL", 1),
            ),
            verify_tls=verify_tls,
        )
        manager = PTZManager(client, self.store, camera)
        hold = pconf.get(self.cfg, "PTZ_HOLD_MAX_S", "2")
        try:
            hold_max_s = float(hold)
        except (TypeError, ValueError):
            hold_max_s = 2.0
        momentary = pconf.geti(self.cfg, "PTZ_MOMENTARY_MS", 400)

        backend = pconf.get(
            self.cfg, "PTZ_MOVE_BACKEND", "onvif").strip().lower()
        if backend not in ("onvif", "isapi", "momentary"):
            print(
                f"WARNING: invalid PTZ_MOVE_BACKEND {backend!r}; using onvif",
                flush=True,
            )
            backend = "onvif"

        onvif_client = None
        onvif_timeout_s = 1
        onvif_gap_s = 0.0
        if backend == "onvif":
            onvif_user, onvif_password = _onvif_credentials_for(camera)
            if not onvif_user or not onvif_password:
                print(
                    f"WARNING: PTZ camera {camera!r} has no ONVIF credentials "
                    "(set PTZ_ONVIF_USER/PTZ_ONVIF_PASS); movement disabled",
                    flush=True,
                )
            else:
                profile = pconf.get(
                    self.cfg, "PTZ_ONVIF_PROFILE", "Profile_1").strip()
                try:
                    timeout_raw = float(pconf.get(
                        self.cfg, "PTZ_ONVIF_TIMEOUT_S", "1"))
                except (TypeError, ValueError):
                    timeout_raw = 1.0
                onvif_timeout_s = max(1, int(round(timeout_raw)))
                if timeout_raw != onvif_timeout_s:
                    print(
                        f"WARNING: PTZ_ONVIF_TIMEOUT_S={timeout_raw:g} rounded "
                        f"to {onvif_timeout_s}s (whole seconds required)",
                        flush=True,
                    )
                try:
                    onvif_gap_s = max(0.0, float(pconf.get(
                        self.cfg, "PTZ_ONVIF_GAP_S", "0")))
                except (TypeError, ValueError):
                    onvif_gap_s = 0.0
                onvif_client = onvif_api.ONVIFClient(
                    host,
                    onvif_user,
                    onvif_password,
                    port=port,
                    scheme=scheme,
                    profile=profile or "Profile_1",
                    verify_tls=verify_tls,
                )

        return PTZService(
            client,
            manager,
            onvif_client=onvif_client,
            move_backend=backend,
            hold_max_s=hold_max_s,
            momentary_ms=momentary,
            onvif_timeout_s=onvif_timeout_s,
            onvif_gap_s=onvif_gap_s,
        )

    async def services(self) -> dict[str, PTZService]:
        async with self._lock:
            if not self._resolved:
                hosts = await self._resolve_hosts()
                for camera, host in hosts.items():
                    try:
                        self._services[camera] = self._build_service(camera, host)
                    except Exception as exc:
                        print(
                            f"WARNING: PTZ camera {camera!r} unavailable: {exc}",
                            flush=True,
                        )
                self._resolved = True
        return self._services

    async def list_cameras(self) -> list[str]:
        return sorted((await self.services()).keys())

    async def move_info(self) -> dict[str, dict]:
        services = await self.services()
        return {
            camera: {
                "backend": service.move_backend,
                "interval_ms": service.move_interval_ms,
            }
            for camera, service in services.items()
        }

    async def get(self, camera: str) -> PTZService:
        services = await self.services()
        service = services.get(camera)
        if service is None:
            raise PresetNotFound(f"PTZ camera {camera!r} is not configured")
        return service

    async def list_presets(
        self, camera: str, include_camera: bool = True
    ) -> list[dict]:
        service = await self.get(camera)
        return await service.list_presets(include_camera=include_camera)

    async def close(self) -> None:
        async with self._lock:
            services = list(self._services.values())
            self._services.clear()
            self._resolved = False
        for service in services:
            await service.aclose()


def manager_from_config(
    cfg: dict,
    *,
    camera: str | None = None,
    host: str | None = None,
    user: str | None = None,
    password: str | None = None,
) -> tuple[PTZClient, PTZManager]:
    """Build a single-camera client + manager (used by the CLI)."""
    names = _camera_names(cfg)
    camera = camera or (names[0] if names else "cam08")
    user = user or _credentials_for(camera)[0]
    password = password or _credentials_for(camera)[1]
    host = host or pconf.get(cfg, "PTZ_HOST", "").strip()
    if not host:
        raise PTZError(
            "no PTZ host: set PTZ_HOST or resolve hosts through the portal "
            "(Frigate /api/config)"
        )
    scheme = pconf.get(cfg, "PTZ_SCHEME", "http").strip().lower()
    if scheme not in ("http", "https"):
        raise PTZError("PTZ_SCHEME must be http or https")
    key = _camera_key(camera)
    client = PTZClient(
        host,
        user,
        password,
        port=pconf.geti(cfg, f"PTZ_PORT_{key}", pconf.geti(cfg, "PTZ_PORT", 80)),
        scheme=scheme,
        channel=pconf.geti(
            cfg, f"PTZ_CHANNEL_{key}", pconf.geti(cfg, "PTZ_CHANNEL", 1)
        ),
        verify_tls=pconf.getb(cfg, "PTZ_VERIFY_TLS", False),
    )
    store = PresetStore(
        pconf.get(cfg, "PTZ_PRESETS_FILE", DEFAULT_PRESETS_FILE),
        legacy_camera=camera,
    )
    return client, PTZManager(client, store, camera)


def _print(data, as_json: bool) -> None:
    if as_json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                print(f"{key}: {json.dumps(value, ensure_ascii=False)}")
            else:
                print(f"{key}: {value}")
    elif isinstance(data, list):
        if not data:
            print("(none)")
            return
        for item in data:
            flags = ""
            if item.get("camera_present") is False:
                flags = "  [MISSING ON CAMERA]"
            print(
                f"{item['name']:<20} camera_id={item['camera_preset_id']:<4} "
                f"updated={item.get('updated_at', '')}{flags}"
                + (f"  # {item['description']}" if item.get("description") else "")
            )
    else:
        print(data)


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Hikvision PTZ control + external preset store.",
    )
    ap.add_argument("--conf", default=DEFAULT_CONF, help="portal.conf path")
    ap.add_argument("--camera", default=None,
                    help="Frigate camera name (default: first PTZ_CAMERA)")
    ap.add_argument("--host", default=None, help="override PTZ_HOST")
    ap.add_argument("--port", type=int, default=None, help="override PTZ_PORT")
    ap.add_argument("--scheme", choices=("http", "https"), default=None)
    ap.add_argument("--channel", type=int, default=None)
    ap.add_argument("--presets-file", default=None, help="override PTZ_PRESETS_FILE")
    ap.add_argument("--user", default=None, help="override camera username")
    ap.add_argument("--pass", dest="password", default=None,
                    help="override camera password (prefer env)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")

    sub = ap.add_subparsers(dest="cmd", required=True)

    def json_opt(parser):
        # The global --json also exists; SUPPRESS keeps a subcommand that does
        # not set it from clobbering the global value.
        parser.add_argument("--json", action="store_true",
                            default=argparse.SUPPRESS, help="machine-readable output")

    health = sub.add_parser("health", help="camera info + preset slots")
    json_opt(health)
    stop = sub.add_parser("stop", help="stop all PTZ movement")
    json_opt(stop)

    pan = sub.add_parser("pan", help="pan continuously for a bounded time")
    pan.add_argument("--speed", type=int, required=True, help="-100..100")
    pan.add_argument("--ms", type=int, default=500)
    json_opt(pan)

    tilt = sub.add_parser("tilt", help="tilt continuously for a bounded time")
    tilt.add_argument("--speed", type=int, required=True, help="-100..100")
    tilt.add_argument("--ms", type=int, default=500)
    json_opt(tilt)

    move = sub.add_parser("move", help="pan + tilt continuously for a bounded time")
    move.add_argument("pan", type=int)
    move.add_argument("tilt", type=int)
    move.add_argument("--ms", type=int, default=500)
    json_opt(move)

    presets = sub.add_parser("presets", help="manage local presets")
    json_opt(presets)
    psub = presets.add_subparsers(dest="preset_cmd", required=True)
    lst = psub.add_parser("list", help="list local presets")
    json_opt(lst)
    clist = psub.add_parser("camera-list", help="list presets stored on the camera")
    json_opt(clist)

    add = psub.add_parser("add", help="save the CURRENT view as a new preset")
    add.add_argument("name")
    add.add_argument("--description", default="")
    add.add_argument("--camera-id", type=int, default=None)
    json_opt(add)

    goto = psub.add_parser("goto", help="move to a stored preset")
    goto.add_argument("name")
    json_opt(goto)

    resave = psub.add_parser("resave", help="overwrite a preset with the CURRENT view")
    resave.add_argument("name")
    json_opt(resave)

    rename = psub.add_parser("rename", help="rename a preset (config only)")
    rename.add_argument("old")
    rename.add_argument("new")
    json_opt(rename)

    describe = psub.add_parser("describe", help="edit a preset description")
    describe.add_argument("name")
    describe.add_argument("--description", required=True)
    json_opt(describe)

    delete = psub.add_parser("delete", help="delete a preset from config + camera")
    delete.add_argument("name")
    json_opt(delete)
    return ap


async def _run(args) -> int:
    cfg, _ = pconf.parse_file(args.conf)
    overrides = (
        ("PTZ_HOST", args.host), ("PTZ_PORT", args.port),
        ("PTZ_SCHEME", args.scheme), ("PTZ_CHANNEL", args.channel),
        ("PTZ_PRESETS_FILE", args.presets_file),
    )
    for key, value in overrides:
        if value is not None:
            cfg[key] = value
    client, manager = manager_from_config(
        cfg, camera=args.camera, host=args.host,
        user=args.user, password=args.password,
    )
    try:
        if args.cmd == "health":
            _print(await client.health(), args.json)
        elif args.cmd == "stop":
            await client.stop()
            _print({"ok": True, "stopped": True}, args.json)
        elif args.cmd == "pan":
            await client.move(args.speed, 0, args.ms)
            _print({"ok": True, "pan": args.speed, "ms": args.ms}, args.json)
        elif args.cmd == "tilt":
            await client.move(0, args.speed, args.ms)
            _print({"ok": True, "tilt": args.speed, "ms": args.ms}, args.json)
        elif args.cmd == "move":
            await client.move(args.pan, args.tilt, args.ms)
            _print({"ok": True, "pan": args.pan, "tilt": args.tilt,
                    "ms": args.ms}, args.json)
        elif args.cmd == "presets":
            cmd = args.preset_cmd
            if cmd == "list":
                _print(await manager.list_presets(), args.json)
            elif cmd == "camera-list":
                _print(await client.camera_presets(), args.json)
            elif cmd == "add":
                _print(await manager.add(
                    args.name, args.description, args.camera_id
                ), args.json)
            elif cmd == "goto":
                _print(await manager.goto(args.name), args.json)
            elif cmd == "resave":
                _print(await manager.resave(args.name), args.json)
            elif cmd == "rename":
                _print(await manager.rename(args.old, args.new), args.json)
            elif cmd == "describe":
                _print(await manager.describe(args.name, args.description),
                       args.json)
            elif cmd == "delete":
                _print(await manager.delete(args.name), args.json)
            else:
                raise PTZError(f"unknown presets command: {cmd}")
        else:
            raise PTZError(f"unknown command: {args.cmd}")
    finally:
        await client.aclose()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except PTZError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
