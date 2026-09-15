#!/usr/bin/env python3
"""Read-only Hikvision ISAPI capability explorer.

STRICTLY READ-ONLY: every request is an HTTP **GET**. Nothing here POSTs, PUTs
or DELETEs, so no device configuration can be changed. Do not add write methods
to this tool.

Targets the firewatch NVR (DS-7616NXI-K2(E) @ 192.168.1.4) which serves ISAPI
over **HTTPS** (self-signed cert) and requires **HTTP Digest** auth. The whole
plain-HTTP interface just 302-redirects to HTTPS, so we always speak TLS.

Usage:
    python3 isapi_explore.py [--host IP] [--user U] [--pass P]
                             [--section system,channels,face,...]
                             [--channel N]        # per-channel Smart configs
                             [--outdir DIR]       # dump raw XML responses
                             [--full]             # print full bodies

Requires: requests
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib3

import requests
from requests.auth import HTTPDigestAuth

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# section -> list of GET paths (all verified readable on this NVR)
SECTIONS: dict[str, list[str]] = {
    "system": [
        "/ISAPI/System/deviceInfo",
        "/ISAPI/System/capabilities",
        "/ISAPI/System/status",
        "/ISAPI/System/time",
    ],
    "network": [
        "/ISAPI/System/Network/interfaces",
    ],
    "channels": [
        "/ISAPI/ContentMgmt/InputProxy/channels",
        "/ISAPI/ContentMgmt/InputProxy/channels/status",
    ],
    "streaming": [
        "/ISAPI/Streaming/channels",
    ],
    "events": [
        "/ISAPI/Event/capabilities",
        "/ISAPI/Event/triggers",
    ],
    "smart": [
        "/ISAPI/Smart/capabilities",
    ],
    "face": [
        "/ISAPI/Intelligent/FDLib/capabilities",
        "/ISAPI/Intelligent/FDLib",
    ],
    "io": [
        "/ISAPI/System/IO/inputs",
        "/ISAPI/System/IO/outputs",
    ],
    "storage": [
        "/ISAPI/ContentMgmt/Storage",
        "/ISAPI/ContentMgmt/Storage/hdd",
    ],
    "recording": [
        "/ISAPI/ContentMgmt/record/tracks",
    ],
}

# per-channel Smart config endpoints (append /<channel>), used by --channel
CHANNEL_SMART = [
    "/ISAPI/Smart/FaceDetect/%d",
    "/ISAPI/Smart/FieldDetection/%d",
    "/ISAPI/Smart/LineDetection/%d",
    "/ISAPI/Smart/RegionEntrance/%d",
    "/ISAPI/Smart/RegionExiting/%d",
]


def _slug(path: str) -> str:
    return path.strip("/").replace("/", "_") or "root"


def get(session: requests.Session, base: str, path: str, outdir: str | None,
        full: bool, preview: int = 600) -> tuple[int, int]:
    url = base + path
    r = session.get(url, timeout=20, verify=False)
    body = r.content or b""
    print(f"\n===== GET {path} =====")
    print(f"http={r.status_code} bytes={len(body)} type={r.headers.get('Content-Type','')}")

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, _slug(path) + ".xml"), "wb") as fh:
            fh.write(body)

    text = body.decode("utf-8", "replace").replace("\r", "")
    if full:
        print(text)
    elif "xml" in (r.headers.get("Content-Type") or ""):
        print(text[:preview] + ("…" if len(text) > preview else ""))
    return r.status_code, len(body)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=os.environ.get("ISAPI_HOST", "192.168.1.4"))
    ap.add_argument("--user", default=os.environ.get("ISAPI_USER", "admin"))
    ap.add_argument("--pass", dest="pwd", default=os.environ.get("ISAPI_PASS", ""))
    ap.add_argument("--section", default="all",
                    help="comma list of: %s (or 'all')" % ",".join(SECTIONS))
    ap.add_argument("--channel", type=int, action="append", default=None,
                    help="also GET the per-channel Smart configs for this channel "
                         "(repeatable)")
    ap.add_argument("--outdir", default=None, help="write raw XML responses here")
    ap.add_argument("--full", action="store_true", help="print full response bodies")
    args = ap.parse_args()

    if not args.pwd:
        print("error: no password (pass --pass or set ISAPI_PASS)", file=sys.stderr)
        return 2

    base = f"https://{args.host}"
    session = requests.Session()
    session.auth = HTTPDigestAuth(args.user, args.pwd)

    wanted = list(SECTIONS) if args.section == "all" else \
        [s.strip() for s in args.section.split(",") if s.strip()]

    print(f"# read-only ISAPI explorer -> {base} as {args.user}")
    total = ok = 0
    for name in wanted:
        paths = SECTIONS.get(name)
        if paths is None:
            print(f"!! unknown section: {name}", file=sys.stderr)
            continue
        print(f"\n########## {name.upper()} ##########")
        for path in paths:
            try:
                code, _ = get(session, base, path, args.outdir, args.full)
                total += 1
                ok += 1 if code == 200 else 0
            except requests.RequestException as exc:
                print(f"\n===== GET {path} =====\n<ERROR: {exc}>")

    for ch in args.channel or []:
        print(f"\n########## SMART CONFIG (channel {ch}) ##########")
        for tmpl in CHANNEL_SMART:
            try:
                total += 1
                get(session, base, tmpl % ch, args.outdir, args.full, preview=300)
            except requests.RequestException as exc:
                print(f"<ERROR: {exc}>")

    print(f"\n# done: {ok}/{total} returned HTTP 200")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
