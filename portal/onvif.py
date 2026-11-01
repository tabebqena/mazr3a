"""Minimal ONVIF PTZ client for the Hikvision camera.

The camera's ISAPI /momentary endpoint ignores <duration> on this firmware, so
manual movement uses ONVIF ContinuousMove with a camera-enforced Timeout. The
portal sends one bounded pulse per refresh; the camera stops itself if the
portal or the Stop request disappears.
"""
from __future__ import annotations

from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape

import httpx

SOAP_NS = "http://www.w3.org/2003/05/soap-envelope"
SCHEMA_NS = "http://www.onvif.org/ver10/schema"
PTZ_NS = "http://www.onvif.org/ver20/ptz/wsdl"
FAULT_TAG = f"{{{SOAP_NS}}}Fault"


class ONVIFError(RuntimeError):
    """Base error for ONVIF PTZ failures."""


class ONVIFAuthError(ONVIFError):
    """The camera rejected the configured ONVIF credentials."""


class ONVIFHTTPError(ONVIFError):
    """The camera returned an unexpected HTTP status."""

    def __init__(self, status_code: int, detail: str = ""):
        self.status_code = status_code
        super().__init__(
            f"camera HTTP {status_code}" + (f": {detail}" if detail else "")
        )


def _envelope(body: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<s:Envelope xmlns:s="{SOAP_NS}" xmlns:tt="{SCHEMA_NS}">'
        f"<s:Body>{body}</s:Body></s:Envelope>"
    )


def _fault_text(text: str) -> str:
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return text.strip()[:300]
    fault = root.find(".//" + FAULT_TAG)
    if fault is None:
        return ""
    return " ".join(fault.itertext()).strip() or "ONVIF fault"


class ONVIFClient:
    """One camera's ONVIF PTZ endpoint."""

    def __init__(
        self,
        host: str,
        user: str,
        password: str,
        *,
        port: int = 80,
        scheme: str = "http",
        profile: str = "Profile_1",
        verify_tls: bool = False,
        timeout: float = 8.0,
    ):
        if not host:
            raise ONVIFError("ONVIF host is empty")
        if not user or not password:
            raise ONVIFError("ONVIF credentials are missing")
        scheme = scheme.lower()
        if scheme not in ("http", "https"):
            scheme = "http"
        host_part = f"[{host}]" if ":" in host and not host.startswith("[") else host
        default_port = 443 if scheme == "https" else 80
        suffix = "" if int(port) == default_port else f":{int(port)}"
        self.base = f"{scheme}://{host_part}{suffix}"
        self.url = self.base + "/onvif/PTZ"
        self.profile = profile
        self._client = httpx.AsyncClient(
            auth=httpx.DigestAuth(user, password),
            verify=verify_tls,
            timeout=timeout,
            follow_redirects=False,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "ONVIFClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def _post(self, body: str) -> httpx.Response:
        try:
            resp = await self._client.post(
                self.url,
                content=body.encode("utf-8"),
                headers={"Content-Type": "application/soap+xml; charset=utf-8"},
            )
        except httpx.HTTPError as exc:
            raise ONVIFError(f"camera unreachable: {exc}") from exc
        if resp.status_code == 401:
            raise ONVIFAuthError(
                "camera rejected the configured ONVIF credentials (401)")
        if resp.status_code != 200:
            raise ONVIFHTTPError(resp.status_code, _fault_text(resp.text))
        fault = _fault_text(resp.text)
        if fault:
            raise ONVIFError(fault)
        return resp

    async def continuous(
        self, pan: int, tilt: int = 0, zoom: int = 0, *, timeout_s: int = 1
    ) -> None:
        for label, value in (("pan", pan), ("tilt", tilt), ("zoom", zoom)):
            if not -100 <= int(value) <= 100:
                raise ONVIFError(f"{label} must be between -100 and 100")
        timeout_s = max(1, int(round(float(timeout_s))))
        body = _envelope(
            f'<tptz:ContinuousMove xmlns:tptz="{PTZ_NS}">'
            f"<tptz:ProfileToken>{escape(self.profile)}</tptz:ProfileToken>"
            "<tptz:Velocity>"
            f'<tt:PanTilt x="{int(pan) / 100:.4f}" '
            f'y="{int(tilt) / 100:.4f}"/>'
            f'<tt:Zoom x="{int(zoom) / 100:.4f}"/>'
            "</tptz:Velocity>"
            f"<tptz:Timeout>PT{timeout_s}S</tptz:Timeout>"
            "</tptz:ContinuousMove>"
        )
        await self._post(body)

    async def stop(self) -> None:
        body = _envelope(
            f'<tptz:Stop xmlns:tptz="{PTZ_NS}">'
            f"<tptz:ProfileToken>{escape(self.profile)}</tptz:ProfileToken>"
            "<tptz:PanTilt>true</tptz:PanTilt>"
            "<tptz:Zoom>true</tptz:Zoom>"
            "</tptz:Stop>"
        )
        await self._post(body)
