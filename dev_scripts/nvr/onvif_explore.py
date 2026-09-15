#!/usr/bin/env python3
"""ONVIF capability explorer for a camera / NVR.

Read-only discovery only: it enumerates the ONVIF services a device exposes and
dumps their capabilities (device, media, PTZ, events, imaging, device-IO,
analytics, recordings, search/replay, receivers). It never changes any setting
on the device.

Built for the firewatch NVR (Hikvision DS-7616NXI-K2(E)) which requires HTTP
Digest auth for ONVIF, so the zeep transport is wired with HTTPDigestAuth.

Device quirks this works around:
  * HTTP Digest (not WS-UsernameToken) is the accepted auth mode;
  * DeviceIO/Recording/Search/Replay/Receiver XAddrs are nested under
    <tt:Extension> and must be registered manually (see
    _register_extension_xaddrs);
  * the whole HTTP interface 302-redirects to HTTPS (self-signed cert), so
    GetServices / snapshots / event PullMessages must be fetched over TLS
    (curl -k). ONVIF SOAP POSTs still work over plain HTTP.

Usage:
    python3 onvif_explore.py [--host IP] [--port N] [--user U] [--pass P]
                             [--section device,media,ptz,events,...]
                             [--json OUT.json]

Defaults target the firewatch NVR; credentials can also come from the
ONVIF_HOST / ONVIF_PORT / ONVIF_USER / ONVIF_PASS environment variables.
Requires the `onvif-zeep` package (pip install onvif-zeep).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Any, Callable


def _wsdl_dir() -> str:
    """Locate the onvif WSDL bundle.

    Depending on how `onvif-zeep` is packaged the `wsdl/` directory sits either
    inside the `onvif` package or as a sibling in site-packages, so probe both.
    """
    import onvif  # noqa: WPS433 (deferred import so --help works without the dep)

    pkg = os.path.dirname(os.path.abspath(onvif.__file__))
    candidates = [
        os.path.join(pkg, "wsdl"),
        os.path.join(os.path.dirname(pkg), "wsdl"),
        "/etc/onvif/wsdl",
    ]
    for cand in candidates:
        if os.path.isfile(os.path.join(cand, "devicemgmt.wsdl")):
            return cand
    raise SystemExit(
        "error: could not find onvif WSDL files; tried: %s" % ", ".join(candidates)
    )


def _plain(obj: Any) -> Any:
    """Convert zeep dataclasses into JSON-friendly Python objects."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    try:
        from zeep.helpers import serialize_object
        return serialize_object(obj)
    except Exception:  # noqa: BLE001
        return str(obj)


def _dump(label: str, obj: Any) -> None:
    print(f"\n===== {label} =====")
    print(json.dumps(_plain(obj), indent=2, default=str, ensure_ascii=False))


def _try(fn: Callable[[], Any]) -> Any:
    try:
        return _plain(fn())
    except Exception as exc:  # noqa: BLE001 - discovery must never abort
        return f"<ERROR: {type(exc).__name__}: {exc}>"


def _svc(cam, name: str) -> Any:
    """Return an ONVIF service handle, creating it on demand.

    ONVIFCamera only pre-creates `devicemgmt` + `event`; everything else is
    built lazily via create_<name>_service().
    """
    return getattr(cam, "create_%s_service" % name)()


def _collect_topics(obj: Any, out: list[str]) -> None:
    """Recursively harvest ONVIF event topic names from a serialized TopicSet."""
    if isinstance(obj, dict):
        for key, val in obj.items():
            if key == "Topic":
                items = val if isinstance(val, list) else [val]
                for it in items:
                    if isinstance(it, str):
                        out.append(it)
                    elif isinstance(it, dict):
                        txt = it.get("_value_1") or it.get("__value__") or it.get("topic")
                        if isinstance(txt, str):
                            out.append(txt)
            else:
                _collect_topics(val, out)
    elif isinstance(obj, list):
        for it in obj:
            _collect_topics(it, out)


def _register_extension_xaddrs(cam) -> None:
    """Populate xaddrs for services nested under capabilities/Extension.

    onvif-zeep's update_xaddrs() only records the TOP-level capability XAddrs
    (Device/Events/Imaging/Media/PTZ). Hikvision reports DeviceIO, Recording,
    Search, Replay and Receiver inside <tt:Extension>, so create_*_service()
    fails with "Device doesn't support service" until we map them here.
    """
    from onvif.definition import SERVICES

    raw = cam.devicemgmt.GetCapabilities({"Category": "All"})
    ext = getattr(raw, "Extension", None)
    elements = getattr(ext, "_value_1", None) or []
    if not isinstance(elements, list):
        elements = [elements]

    want = {
        "DeviceIO": "deviceio",
        "Recording": "recording",
        "Search": "search",
        "Replay": "replay",
        "Receiver": "receiver",
    }
    for el in elements:
        name = str(getattr(el, "tag", "") or "").split("}")[-1]
        svc = want.get(name)
        if not svc:
            continue
        xaddr = el.find(".//{http://www.onvif.org/ver10/schema}XAddr")
        if xaddr is not None and xaddr.text:
            cam.xaddrs[SERVICES[svc]["ns"]] = xaddr.text


def build_camera(host: str, port: int, user: str, pwd: str):
    """Build an ONVIFCamera wired for HTTP Digest (Hikvision/HiWatch needs it)."""
    import requests
    from requests.auth import HTTPDigestAuth
    from zeep.transports import Transport
    from onvif import ONVIFCamera

    session = requests.Session()
    session.auth = HTTPDigestAuth(user, pwd)
    transport = Transport(session=session, timeout=30, operation_timeout=90)

    cam = ONVIFCamera(host, port, user, pwd, _wsdl_dir(), transport=transport)
    # The constructor already refreshed xaddrs; re-run defensively.
    try:
        cam.update_xaddrs()
    except Exception as exc:  # noqa: BLE001
        print(f"# note: update_xaddrs hit {type(exc).__name__}: {exc}", file=sys.stderr)
    try:
        _register_extension_xaddrs(cam)
    except Exception as exc:  # noqa: BLE001
        print(f"# note: extension xaddrs failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    return cam


def section_device(cam, out: dict) -> None:
    out["device_information"] = _try(lambda: cam.devicemgmt.GetDeviceInformation())
    out["capabilities"] = _try(lambda: cam.devicemgmt.GetCapabilities({"Category": "All"}))
    out["services"] = _try(lambda: cam.devicemgmt.GetServices({"IncludeCapability": False}))
    out["system_date_time"] = _try(lambda: cam.devicemgmt.GetSystemDateAndTime())
    out["network_interfaces"] = _try(lambda: cam.devicemgmt.GetNetworkInterfaces())
    out["users"] = _try(lambda: cam.devicemgmt.GetUsers())

    _dump("DeviceInformation", out["device_information"])
    _dump("Services", out["services"])
    _dump("Users", out["users"])


def section_media(cam, out: dict) -> None:
    media = _svc(cam, "media")
    profiles = _try(lambda: media.GetProfiles())
    out["media_profiles"] = profiles
    _dump("MediaProfiles", profiles)

    xaddrs: dict[str, str] = {}
    if isinstance(out.get("capabilities"), dict):
        pass

    uris = []
    if isinstance(profiles, list):
        for prof in profiles:
            token = prof.get("token") if isinstance(prof, dict) else None
            if not token:
                continue
            stream = _try(lambda t=token: media.GetStreamUri({
                "StreamSetup": {
                    "Stream": "RTP-Unicast",
                    "Transport": {"Protocol": "RTSP"},
                },
                "ProfileToken": t,
            }))
            snap = _try(lambda t=token: media.GetSnapshotUri({"ProfileToken": t}))
            uris.append({
                "token": token,
                "name": prof.get("Name") if isinstance(prof, dict) else None,
                "stream_uri": stream,
                "snapshot_uri": snap,
            })
    out["stream_uris"] = uris
    _dump("StreamUris", uris)

    out["media_service_cap"] = _try(lambda: media.GetServiceCapabilities())
    out["video_sources"] = _try(lambda: media.GetVideoSources())
    out["video_source_configs"] = _try(lambda: media.GetVideoSourceConfigurations())
    _dump("VideoSources", out["video_sources"])


def section_ptz(cam, out: dict) -> None:
    ptz = _svc(cam, "ptz")
    out["ptz_configurations"] = _try(lambda: ptz.GetConfigurations())
    out["ptz_nodes"] = _try(lambda: ptz.GetNodes())
    out["ptz_service_cap"] = _try(lambda: ptz.GetServiceCapabilities())
    _dump("PTZConfigurations", out["ptz_configurations"])
    _dump("PTZNodes", out["ptz_nodes"])


def section_events(cam, out: dict) -> None:
    events = _svc(cam, "events")
    props = _try(lambda: events.GetEventProperties())
    out["event_properties"] = props

    topics: list[str] = []
    _collect_topics(props, topics)
    out["event_topics"] = topics
    _dump(f"EventTopics ({len(topics)})", topics)

    out["event_service_cap"] = _try(lambda: events.GetServiceCapabilities())
    _dump("EventServiceCapabilities", out["event_service_cap"])


def section_imaging(cam, out: dict) -> None:
    imaging = _svc(cam, "imaging")
    out["imaging_service_cap"] = _try(lambda: imaging.GetServiceCapabilities())
    out["imaging_sources"] = _try(lambda: imaging.GetSources())
    out["imaging_options"] = _try(lambda: imaging.GetOptions({"VideoSourceToken": "1"}))
    _dump("ImagingServiceCapabilities", out["imaging_service_cap"])
    _dump("ImagingSources", out["imaging_sources"])


def section_deviceio(cam, out: dict) -> None:
    try:
        devio = _svc(cam, "deviceio")
    except Exception as exc:  # noqa: BLE001
        out["deviceio_error"] = f"{type(exc).__name__}: {exc}"
        _dump("DeviceIO", out["deviceio_error"])
        return
    out["video_outputs"] = _try(lambda: devio.GetVideoOutputs())
    out["relay_outputs"] = _try(lambda: devio.GetRelayOutputs())
    out["deviceio_service_cap"] = _try(lambda: devio.GetServiceCapabilities())
    _dump("VideoOutputs", out["video_outputs"])
    _dump("RelayOutputs", out["relay_outputs"])


def section_analytics(cam, out: dict) -> None:
    try:
        analytics = _svc(cam, "analytics")
    except Exception as exc:  # noqa: BLE001
        out["analytics_error"] = f"{type(exc).__name__}: {exc}"
        _dump("Analytics", out["analytics_error"])
        return
    out["analytics_modules"] = _try(lambda: analytics.GetAnalyticsModules({
        "ConfigurationToken": "0",
    }))
    out["analytics_service_cap"] = _try(lambda: analytics.GetServiceCapabilities())
    _dump("AnalyticsServiceCapabilities", out["analytics_service_cap"])


def section_recording(cam, out: dict) -> None:
    rec = _svc(cam, "recording")
    out["recordings"] = _try(lambda: rec.GetRecordings())
    out["recording_cap"] = _try(lambda: rec.GetServiceCapabilities())
    _dump("Recordings", out["recordings"])


def section_search(cam, out: dict) -> None:
    srv = _svc(cam, "search")
    out["search_cap"] = _try(lambda: srv.GetServiceCapabilities())
    _dump("SearchServiceCapabilities", out["search_cap"])


def section_replay(cam, out: dict) -> None:
    srv = _svc(cam, "replay")
    out["replay_cap"] = _try(lambda: srv.GetServiceCapabilities())
    _dump("ReplayServiceCapabilities", out["replay_cap"])


def section_receiver(cam, out: dict) -> None:
    srv = _svc(cam, "receiver")
    out["receivers"] = _try(lambda: srv.GetReceivers())
    out["receiver_cap"] = _try(lambda: srv.GetServiceCapabilities())
    _dump("Receivers", out["receivers"])


SECTIONS: dict[str, Callable[[Any, dict], None]] = {
    "device": section_device,
    "media": section_media,
    "ptz": section_ptz,
    "events": section_events,
    "imaging": section_imaging,
    "deviceio": section_deviceio,
    "analytics": section_analytics,
    "recording": section_recording,
    "search": section_search,
    "replay": section_replay,
    "receiver": section_receiver,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=os.environ.get("ONVIF_HOST", "192.168.1.4"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("ONVIF_PORT", "80")))
    ap.add_argument("--user", default=os.environ.get("ONVIF_USER", "admin"))
    ap.add_argument("--pass", dest="pwd", default=os.environ.get("ONVIF_PASS", ""))
    ap.add_argument("--section", default="all",
                    help="comma list of: %s (or 'all')" % ",".join(SECTIONS))
    ap.add_argument("--json", dest="json_out", default=None,
                    help="write the collected data as JSON to this path")
    args = ap.parse_args()

    if not args.pwd:
        print("error: no password (pass --pass or set ONVIF_PASS)", file=sys.stderr)
        return 2

    cam = build_camera(args.host, args.port, args.user, args.pwd)
    print(f"# ONVIF explorer -> {args.host}:{args.port} as {args.user}")

    wanted = list(SECTIONS) if args.section == "all" else \
        [s.strip() for s in args.section.split(",") if s.strip()]

    out: dict[str, Any] = {"host": args.host, "port": args.port}
    for name in wanted:
        fn = SECTIONS.get(name)
        if fn is None:
            print(f"!! unknown section: {name}", file=sys.stderr)
            continue
        print(f"\n########## {name.upper()} ##########")
        try:
            fn(cam, out)
        except Exception:  # noqa: BLE001
            traceback.print_exc()

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, default=str, ensure_ascii=False)
        print(f"\n# wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
