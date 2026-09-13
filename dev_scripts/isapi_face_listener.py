#!/usr/bin/env python3
"""Read-only Hikvision ISAPI alert-stream listener (face capture).

STRICTLY READ-ONLY: opens ONE streaming **GET** to
/ISAPI/Event/notification/alertStream and reads whatever the NVR pushes. It
never POSTs/PUTs/DELETEs and changes nothing on the device.

It parses the `multipart/mixed` stream into parts, summarises each XML event
(type/channel/time/face fields) and saves any JPEG face crops. Every raw byte is
also teed to `<outdir>/_raw_stream.bin` for offline examination.

Usage:
    python3 isapi_face_listener.py [--host IP] [--user U] [--pass P]
                                   [--seconds 90] [--max-events N]
                                   [--outdir /tmp/facecap] [--full-xml]

Transport: HTTPS + HTTP Digest (self-signed cert). Requires: requests
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time

import requests
import urllib3
from requests.auth import HTTPDigestAuth

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ALERT_PATH = "/ISAPI/Event/notification/alertStream"
FIELDS = ("eventType", "channelID", "dateTime", "eventState", "eventDescription",
          "activePostCount", "ipAddress", "pictureURL", "FaceRect", "faceScore")


def _xml_field(text: str, name: str) -> str | None:
    m = re.search(rf"<(?:\w+:)?{name}>([^<]*)</(?:\w+:)?{name}>", text, re.I)
    return m.group(1).strip() if m else None


def _boundary(content_type: str) -> bytes | None:
    m = re.search(r'boundary="?([^";]+)"?', content_type or "")
    return m.group(1).encode() if m else None


def _summarise_xml(text: str, outdir: str, idx: int, full: bool) -> dict:
    info = {f: _xml_field(text, f) for f in FIELDS}
    info = {k: v for k, v in info.items() if v}
    et = info.get("eventType", "?")
    ch = info.get("channelID", "?")
    dt = info.get("dateTime", "?")
    print(f"  [{idx:02d}] XML  eventType={et:<18} ch={ch:<3} time={dt} "
          f"state={info.get('eventState','?')} url={info.get('pictureURL','-')}")
    with open(os.path.join(outdir, f"{idx:02d}-{et}-ch{ch}.xml"), "w",
              encoding="utf-8") as fh:
        fh.write(text)
    if full:
        print(text)
    return info


def fetch_picture(session, url: str, outdir: str, counters: dict, base: str) -> None:
    """GET a face picture referenced by a pictureURL (read-only)."""
    if url.startswith("/"):
        url = base + url
    try:
        r = session.get(url, verify=False, timeout=20)
    except requests.RequestException as exc:
        print(f"  -> picture fetch failed: {exc}")
        return
    data = r.content or b""
    if r.status_code == 200 and data[:3] == b"\xff\xd8\xff":
        counters["images"] += 1
        ts = time.strftime("%Y%m%d-%H%M%S")
        fname = os.path.join(outdir, f"{ts}-face{counters['images']:02d}.jpg")
        with open(fname, "wb") as fh:
            fh.write(data)
        print(f"  -> picture {len(data)}B -> {os.path.basename(fname)}")
    else:
        print(f"  -> pictureURL http={r.status_code} bytes={len(data)} (not jpeg)")


def handle_part(part: bytes, outdir: str, counters: dict, full: bool,
                session=None, base: str = "") -> None:
    # part = headers \r\n\r\n body ; strip the leading CRLF after the boundary
    if part.startswith(b"\r\n"):
        part = part[2:]
    if not part or part.startswith(b"--"):
        return
    head, _, body = part.partition(b"\r\n\r\n")
    ctype = ""
    for line in head.decode("latin-1").splitlines():
        if line.lower().startswith("content-type:"):
            ctype = line.split(":", 1)[1].strip().lower()

    counters["parts"] += 1
    if "image" in ctype or body[:3] == b"\xff\xd8\xff":
        counters["images"] += 1
        idx = counters["images"]
        ts = time.strftime("%Y%m%d-%H%M%S")
        fname = os.path.join(outdir, f"{ts}-img{idx:02d}.jpg")
        with open(fname, "wb") as fh:
            fh.write(body)
        print(f"  [{counters['parts']:02d}] IMG  {ctype or 'jpeg'} "
              f"{len(body)}B -> {os.path.basename(fname)}")
        return

    text = body.decode("utf-8", "replace")
    if "xml" in ctype or "<EventNotificationAlert" in text or "<EventNotification" in text:
        info = _summarise_xml(text, outdir, counters["parts"], full)
        et = info.get("eventType", "?")
        counters["by_type"][et] = counters["by_type"].get(et, 0) + 1
        counters["events"] += 1
        purl = info.get("pictureURL")
        if purl and session is not None:
            fetch_picture(session, purl, outdir, counters, base)
    else:
        counters["other"] += 1
        print(f"  [{counters['parts']:02d}] OTHER ctype={ctype!r} bytes={len(body)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=os.environ.get("ISAPI_HOST", "192.168.1.4"))
    ap.add_argument("--user", default=os.environ.get("ISAPI_USER", "admin"))
    ap.add_argument("--pass", dest="pwd", default=os.environ.get("ISAPI_PASS", ""))
    ap.add_argument("--seconds", type=int, default=60,
                    help="max capture window (default 60)")
    ap.add_argument("--max-events", type=int, default=0,
                    help="stop after this many events (0 = no limit)")
    ap.add_argument("--outdir", default="/tmp/facecap")
    ap.add_argument("--full-xml", action="store_true", help="print full event XML")
    args = ap.parse_args()

    if not args.pwd:
        print("error: no password (--pass or ISAPI_PASS)", file=sys.stderr)
        return 2

    os.makedirs(args.outdir, exist_ok=True)
    raw_path = os.path.join(args.outdir, "_raw_stream.bin")
    base = f"https://{args.host}"
    url = base + ALERT_PATH

    session = requests.Session()
    session.auth = HTTPDigestAuth(args.user, args.pwd)
    session.headers["Accept"] = "multipart/mixed"

    counters = {"parts": 0, "events": 0, "images": 0, "other": 0, "by_type": {}}
    deadline = time.time() + args.seconds
    print(f"# listening {url} for up to {args.seconds}s -> {args.outdir}")

    buf = bytearray()
    boundary = None
    try:
        with session.get(url, stream=True, verify=False,
                         timeout=(15, args.seconds + 10)) as resp:
            print(f"# http={resp.status_code} type={resp.headers.get('Content-Type','')}")
            if resp.status_code != 200:
                print(resp.text[:500])
                return 1
            boundary = _boundary(resp.headers.get("Content-Type", "")) or b"boundary"
            print(f"# boundary={boundary!r}")

            with open(raw_path, "wb") as raw:
                for chunk in resp.iter_content(chunk_size=4096):
                    if not chunk:
                        if time.time() > deadline:
                            break
                        continue
                    raw.write(chunk)
                    buf.extend(chunk)
                    delim = b"--" + boundary
                    while True:
                        i = buf.find(delim)
                        if i == -1:
                            break
                        j = buf.find(delim, i + len(delim))
                        if j == -1:
                            break
                        part = bytes(buf[i + len(delim):j])
                        del buf[:j]
                        handle_part(part, args.outdir, counters, args.full_xml,
                                    session=session, base=base)
                    if args.max_events and counters["images"] + counters["events"] >= args.max_events:
                        print("# reached --max-events, stopping")
                        break
                    if time.time() > deadline:
                        print("# time window elapsed, stopping")
                        break
    except requests.RequestException as exc:
        print(f"# stream ended: {type(exc).__name__}: {exc}")

    print(f"\n# SUMMARY parts={counters['parts']} events={counters['events']} "
          f"images={counters['images']} other={counters['other']}")
    if counters["by_type"]:
        for et, n in sorted(counters["by_type"].items(), key=lambda x: -x[1]):
            print(f"    {n:>4}  {et}")
    print(f"# raw stream saved to {raw_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
