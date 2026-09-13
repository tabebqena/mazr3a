#!/usr/bin/env python3
"""Read-only Hikvision ISAPI event-stream listener (general, stores raw data).

STRICTLY READ-ONLY: it opens streaming **GET** requests to an ISAPI event
stream (default `/ISAPI/Event/notification/alertStream`) and records everything
the NVR pushes. It never POSTs/PUTs/DELETEs, so no device config can change.

It is deliberately *general*: it does not filter for faces. Every multipart part
is stored **raw**, and an index of all events is written as JSONL.

What it stores (all under --outdir):

    _stream.bin            every raw byte read from the HTTP response
    parts/NNNNNN-<t>.<ext> each multipart part, raw (headers + body)
    pictures/*.jpg         any image/jpeg part (or fetched <pictureURL>)
    events.jsonl           one JSON object per part: content-type, bytes,
                           file, and (for XML) eventType/channelID/dateTime/
                           eventState/eventDescription/activePostCount

Usage:
    python3 isapi_alert_listener.py [--host IP] [--user U] [--pass P]
        [--stream /ISAPI/Event/notification/alertStream]
        [--seconds 0]        # 0 = run forever (Ctrl-C to stop)
        [--max-parts 0]      # 0 = unlimited
        [--outdir /tmp/isapi_events]
        [--reconnect]        # reopen if the stream drops
        [--fetch-pictures]   # also GET any <pictureURL> found in XML parts
        [--print-xml]        # echo XML bodies to stdout

Transport: HTTPS + HTTP Digest (self-signed cert). Requires: requests
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import time

import requests
import urllib3
from requests.auth import HTTPDigestAuth

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DEFAULT_STREAM = "/ISAPI/Event/notification/alertStream"
# fields we try to lift out of an XML EventNotificationAlert (best-effort)
XML_FIELDS = ("eventType", "channelID", "channelName", "dateTime", "eventState",
              "eventDescription", "activePostCount", "ipAddress", "macAddress",
              "serialNo", "pictureURL", "faceScore", "FaceRect")

_stop = False


def _on_sigint(signum, frame):  # noqa: ARG001
    global _stop
    _stop = True
    print("\n# interrupt received, finishing current read…", file=sys.stderr)


def _xml_field(text: str, name: str) -> str | None:
    m = re.search(rf"<(?:\w+:)?{name}>([^<]*)</(?:\w+:)?{name}>", text, re.I)
    return m.group(1).strip() if m else None


def _boundary(content_type: str) -> bytes | None:
    m = re.search(r'boundary="?([^";]+)"?', content_type or "")
    return m.group(1).encode() if m else None


def _ext_for(content_type: str, body: bytes) -> str:
    ct = (content_type or "").lower()
    if "jpeg" in ct or "jpg" in ct or body[:3] == b"\xff\xd8\xff":
        return "jpg"
    if "png" in ct or body[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if "xml" in ct or body.lstrip()[:1] == b"<":
        return "xml"
    if "text" in ct:
        return "txt"
    return "bin"


class Recorder:
    def __init__(self, outdir: str, session, base: str, fetch_pictures: bool,
                 print_xml: bool):
        self.outdir = outdir
        self.parts_dir = os.path.join(outdir, "parts")
        self.pics_dir = os.path.join(outdir, "pictures")
        os.makedirs(self.parts_dir, exist_ok=True)
        os.makedirs(self.pics_dir, exist_ok=True)
        self.stream_path = os.path.join(outdir, "_stream.bin")
        self.index_path = os.path.join(outdir, "events.jsonl")
        self.session = session
        self.base = base
        self.fetch_pictures = fetch_pictures
        self.print_xml = print_xml
        self.n_parts = 0
        self.by_type: dict[str, int] = {}
        self.n_pictures = 0
        self._stream_fh = open(self.stream_path, "ab")
        self._index_fh = open(self.index_path, "a", encoding="utf-8", buffering=1)

    def close(self):
        self._stream_fh.close()
        self._index_fh.close()

    def tee(self, chunk: bytes):
        self._stream_fh.write(chunk)

    def handle_part(self, part: bytes) -> None:
        # strip the CRLF that follows the boundary line
        if part.startswith(b"\r\n"):
            part = part[2:]
        if not part:
            return
        if part.startswith(b"--"):  # final "--boundary--"
            return
        head, _, body = part.partition(b"\r\n\r\n")
        headers = head.decode("latin-1", "replace")
        ctype = ""
        for line in headers.splitlines():
            if line.lower().startswith("content-type:"):
                ctype = line.split(":", 1)[1].strip()

        self.n_parts += 1
        ext = _ext_for(ctype, body)
        record: dict = {
            "part": self.n_parts,
            "ts_host": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "content_type": ctype,
            "bytes": len(body),
        }

        if ext in ("jpg", "png"):
            fname = os.path.join(self.pics_dir,
                                 f"{self.n_parts:06d}-{time.strftime('%Y%m%d-%H%M%S')}.{ext}")
        else:
            base = "part"
            if ext == "xml":
                text = body.decode("utf-8", "replace")
                fields = {f: _xml_field(text, f) for f in XML_FIELDS}
                fields = {k: v for k, v in fields.items() if v}
                record.update(fields)
                et = fields.get("eventType", "event")
                base = re.sub(r"[^A-Za-z0-9_.-]", "_", et)[:40] or "xml"
                self.by_type[et] = self.by_type.get(et, 0) + 1
                if self.print_xml:
                    print(text)
                purl = fields.get("pictureURL")
                if purl and self.fetch_pictures:
                    self._fetch_picture(purl)
            fname = os.path.join(self.parts_dir, f"{self.n_parts:06d}-{base}.{ext}")

        with open(fname, "wb") as fh:
            fh.write(part)  # RAW part incl. headers, as received
        record["file"] = os.path.relpath(fname, self.outdir)
        self._index_fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        et = record.get("eventType", f"<{ext}>")
        ch = record.get("channelID", "-")
        st = record.get("eventState", "-")
        print(f"  [{self.n_parts:06d}] {et:<20} ch={ch:<3} state={st:<8} "
              f"{record['bytes']:>7}B -> {record['file']}")

    def _fetch_picture(self, url: str) -> None:
        if url.startswith("/"):
            url = self.base + url
        try:
            r = self.session.get(url, verify=False, timeout=20)
        except requests.RequestException as exc:
            print(f"        pictureURL fetch failed: {exc}")
            return
        data = r.content or b""
        if r.status_code == 200 and data[:3] == b"\xff\xd8\xff":
            self.n_pictures += 1
            fname = os.path.join(self.pics_dir,
                                 f"{self.n_parts:06d}-url-{self.n_pictures:04d}.jpg")
            with open(fname, "wb") as fh:
                fh.write(data)
            print(f"        pictureURL {len(data)}B -> pictures/{os.path.basename(fname)}")


def run_once(rec: Recorder, args, url: str) -> bool:
    """One connection. Returns True if it ended cleanly (caller may reconnect)."""
    with rec.session.get(url, stream=True, verify=False,
                         timeout=(15, 120)) as resp:
        print(f"# connected http={resp.status_code} "
              f"type={resp.headers.get('Content-Type','')}")
        if resp.status_code != 200:
            print(resp.text[:400])
            return False
        boundary = _boundary(resp.headers.get("Content-Type", "")) or b"boundary"
        print(f"# boundary={boundary!r}")
        buf = bytearray()
        delim = b"--" + boundary
        deadline = None if args.seconds <= 0 else time.time() + args.seconds

        for chunk in resp.iter_content(chunk_size=4096):
            if _stop:
                return True
            if chunk:
                rec.tee(chunk)
                buf.extend(chunk)
                while True:
                    i = buf.find(delim)
                    if i == -1:
                        break
                    j = buf.find(delim, i + len(delim))
                    if j == -1:
                        break
                    rec.handle_part(bytes(buf[i + len(delim):j]))
                    del buf[:j]
                if args.max_parts and rec.n_parts >= args.max_parts:
                    print("# reached --max-parts")
                    return True
            if deadline and time.time() > deadline:
                print("# time window elapsed")
                return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=os.environ.get("ISAPI_HOST", "192.168.1.4"))
    ap.add_argument("--user", default=os.environ.get("ISAPI_USER", "admin"))
    ap.add_argument("--pass", dest="pwd", default=os.environ.get("ISAPI_PASS", ""))
    ap.add_argument("--stream", default=DEFAULT_STREAM,
                    help="ISAPI stream path (default: alertStream)")
    ap.add_argument("--seconds", type=int, default=0,
                    help="capture window; 0 = run forever (Ctrl-C to stop)")
    ap.add_argument("--max-parts", type=int, default=0, help="0 = unlimited")
    ap.add_argument("--outdir", default="/tmp/isapi_events")
    ap.add_argument("--reconnect", action="store_true",
                    help="reopen the stream if it drops")
    ap.add_argument("--fetch-pictures", action="store_true",
                    help="GET any <pictureURL> found in XML parts")
    ap.add_argument("--print-xml", action="store_true", help="echo XML bodies")
    args = ap.parse_args()

    if not args.pwd:
        print("error: no password (--pass or ISAPI_PASS)", file=sys.stderr)
        return 2

    signal.signal(signal.SIGINT, _on_sigint)
    base = f"https://{args.host}"
    url = base + args.stream
    session = requests.Session()
    session.auth = HTTPDigestAuth(args.user, args.pwd)
    session.headers["Accept"] = "multipart/mixed"

    os.makedirs(args.outdir, exist_ok=True)
    rec = Recorder(args.outdir, session, base, args.fetch_pictures, args.print_xml)

    print(f"# read-only ISAPI listener: {url}")
    print(f"# outdir={args.outdir}  window={'∞' if args.seconds<=0 else str(args.seconds)+'s'}"
          f"  reconnect={args.reconnect}")
    try:
        while True:
            try:
                clean = run_once(rec, args, url)
            except requests.RequestException as exc:
                print(f"# stream error: {type(exc).__name__}: {exc}")
                clean = False
            if _stop or (args.seconds > 0 and not args.reconnect):
                break
            if args.max_parts and rec.n_parts >= args.max_parts:
                break
            if not args.reconnect:
                break
            if not clean:
                time.sleep(3)
    finally:
        rec.close()

    print(f"\n# SUMMARY parts={rec.n_parts} pictures={rec.n_pictures} "
          f"types={len(rec.by_type)}")
    for et, n in sorted(rec.by_type.items(), key=lambda x: -x[1]):
        print(f"    {n:>5}  {et}")
    print(f"# raw stream : {rec.stream_path}")
    print(f"# index      : {rec.index_path}")
    print(f"# parts dir  : {rec.parts_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
